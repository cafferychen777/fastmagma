#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static FILE *open_path(PyObject *path, const char *mode) {
#ifdef _WIN32
    PyObject *decoded = NULL;
    if (!PyUnicode_FSDecoder(path, &decoded)) return NULL;
    wchar_t *wide = PyUnicode_AsWideCharString(decoded, NULL);
    Py_DECREF(decoded);
    if (!wide) return NULL;
    FILE *stream = _wfopen(wide, mode[0] == 'r' ? L"rb" : L"wb");
    PyMem_Free(wide);
    return stream;
#else
    return fopen(PyBytes_AS_STRING(path), mode);
#endif
}

/* Filter before constructing Python strings for numeric columns. Numeric QC
 * remains in io.py; non-plain tables are delegated to the pandas parser. */
static int space(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\v' || c == '\f';
}

typedef struct {
    unsigned char buffer[65536];
    size_t start, end;
} Reader;

static int read_line(FILE *stream, Reader *reader, char **line, size_t *capacity, size_t *length) {
    *length = 0;
    for (;;) {
        if (reader->start == reader->end) {
            reader->end = fread(reader->buffer, 1, sizeof(reader->buffer), stream);
            reader->start = 0;
            if (!reader->end) {
                if (ferror(stream)) { PyErr_SetFromErrno(PyExc_OSError); return -1; }
                return *length != 0;
            }
        }
        unsigned char *begin = reader->buffer + reader->start;
        size_t available = reader->end - reader->start;
        unsigned char *newline = memchr(begin, '\n', available);
        size_t count = newline ? (size_t)(newline - begin) + 1 : available;
        if (memchr(begin, 0, count)) return 2;
        if (*capacity < *length + count) {
            size_t next = (*length + count) * 2;
            char *resized = realloc(*line, next);
            if (!resized) { PyErr_NoMemory(); return -1; }
            *line = resized;
            *capacity = next;
        }
        memcpy(*line + *length, begin, count);
        *length += count;
        reader->start += count;
        if (newline) return 1;
    }
}

static PyObject *filter_pval(PyObject *self, PyObject *args) {
    PyObject *source_obj, *destination_obj, *snps, *source = NULL, *destination = NULL;
    PyObject *identifiers = NULL, *iterator = NULL, *item = NULL, *result = NULL;
    FILE *input = NULL, *output = NULL;
    char *line = NULL;
    size_t capacity = 0, length = 0;
    Py_ssize_t rows = 0;
    int header_fields = -1, snp_column = -1, status;
    Reader reader = {{0}, 0, 0};
    (void)self;
    if (!PyArg_ParseTuple(args, "OOO:filter_pval", &source_obj, &snps, &destination_obj)) return NULL;
    if (!PyUnicode_FSConverter(source_obj, &source) ||
        !PyUnicode_FSConverter(destination_obj, &destination)) goto cleanup;
    int shared_index = PyDict_Check(snps);
    if (shared_index) {
        identifiers = Py_NewRef(snps);
    } else {
        identifiers = PySet_New(NULL);
        iterator = PyObject_GetIter(snps);
        if (!identifiers || !iterator) goto cleanup;
        while ((item = PyIter_Next(iterator))) {
            PyObject *encoded = PyUnicode_AsUTF8String(item);
            Py_CLEAR(item);
            if (!encoded) goto cleanup;
            int added = PySet_Add(identifiers, encoded);
            Py_DECREF(encoded);
            if (added < 0) goto cleanup;
        }
        if (PyErr_Occurred()) goto cleanup;
    }
    input = open_path(source, "rb");
    if (!input) {
        if (!PyErr_Occurred()) PyErr_SetFromErrnoWithFilenameObject(PyExc_OSError, source_obj);
        goto cleanup;
    }
    output = open_path(destination, "wb");
    if (!output) {
        if (!PyErr_Occurred()) PyErr_SetFromErrnoWithFilenameObject(PyExc_OSError, destination_obj);
        goto cleanup;
    }
    while ((status = read_line(input, &reader, &line, &capacity, &length)) > 0) {
        if (status == 2 || memchr(line, '"', length) ||
            memchr(line, '\v', length) || memchr(line, '\f', length)) goto fallback;
        for (size_t i = 0; i < length; i++) {
            if ((unsigned char)line[i] >= 128 ||
                (line[i] == '\r' && (i + 1 == length || line[i + 1] != '\n'))) goto fallback;
        }
        int fields = 0, p_columns = 0, n_columns = 0, snp_columns = 0;
        const char *snp = NULL;
        size_t snp_length = 0;
        for (size_t start = 0; start < length;) {
            while (start < length && space((unsigned char)line[start])) start++;
            if (start == length) break;
            size_t end = start;
            while (end < length && !space((unsigned char)line[end])) end++;
            if (header_fields < 0) {
                if (end - start == 3 && !memcmp(line + start, "SNP", 3)) {
                    snp_column = fields;
                    snp_columns++;
                }
                if (end - start == 1 && line[start] == 'P') p_columns++;
                if (end - start == 1 && line[start] == 'N') n_columns++;
            } else if (fields == snp_column) {
                snp = line + start;
                snp_length = end - start;
            }
            fields++;
            start = end;
        }
        if (!fields) continue;
        if (header_fields < 0) {
            if (snp_columns != 1 || p_columns != 1 || n_columns != 1) goto fallback;
            header_fields = fields;
        } else {
            if (fields != header_fields || !snp) goto fallback;
            rows++;
            if ((rows & 65535) == 0 && PyErr_CheckSignals() < 0) goto cleanup;
            PyObject *key = shared_index
                ? PyUnicode_DecodeASCII(snp, (Py_ssize_t)snp_length, NULL)
                : PyBytes_FromStringAndSize(snp, (Py_ssize_t)snp_length);
            if (!key) goto cleanup;
            int present = shared_index ? PyDict_Contains(identifiers, key) : PySet_Contains(identifiers, key);
            Py_DECREF(key);
            if (present < 0) goto cleanup;
            if (!present) continue;
        }
        if (fwrite(line, 1, length, output) != length) {
            PyErr_SetFromErrno(PyExc_OSError);
            goto cleanup;
        }
    }
    if (status < 0) goto cleanup;
    if (header_fields < 0) goto fallback;
    result = PyLong_FromSsize_t(rows);
    goto cleanup;
fallback:
    result = Py_NewRef(Py_None);
cleanup:
    free(line);
    if (input) fclose(input);
    if (output && fclose(output) != 0 && result) {
        Py_CLEAR(result);
        PyErr_SetFromErrno(PyExc_OSError);
    }
    Py_XDECREF(source);
    Py_XDECREF(destination);
    Py_XDECREF(identifiers);
    Py_XDECREF(iterator);
    Py_XDECREF(item);
    return result;
}

/* CPython's locale-independent converter is also used by float(str). Requiring
 * the complete ASCII-trimmed token rejects underscores and hexadecimal forms. */
static PyObject *parse_numeric(PyObject *self, PyObject *args) {
    PyObject *values, *destination, *iterator = NULL, *item = NULL;
    Py_buffer output = {0};
    Py_ssize_t index = 0;
    (void)self;
    if (!PyArg_ParseTuple(args, "OO:parse_numeric", &values, &destination)) return NULL;
    if (PyObject_GetBuffer(destination, &output, PyBUF_WRITABLE | PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES) < 0) return NULL;
    if (output.ndim != 1 || output.itemsize != sizeof(double) ||
        !output.format || (strcmp(output.format, "d") && strcmp(output.format, "=d")) ||
        !PyBuffer_IsContiguous(&output, 'C')) {
        PyErr_SetString(PyExc_ValueError, "Output must be a contiguous writable native float64 vector");
        goto error;
    }
    iterator = PyObject_GetIter(values);
    if (!iterator) goto error;
    while ((item = PyIter_Next(iterator))) {
        if (index >= output.len / (Py_ssize_t)sizeof(double)) {
            PyErr_SetString(PyExc_ValueError, "Output length differs from input length");
            goto error;
        }
        Py_ssize_t length;
        double value = Py_NAN;
        const char *text = PyUnicode_AsUTF8AndSize(item, &length);
        if (!text) {
            if (!PyErr_ExceptionMatches(PyExc_UnicodeEncodeError)) goto error;
            PyErr_Clear();
            goto store_value;
        }
        const char *stop = text + length;
        while (text < stop && space((unsigned char)*text)) text++;
        while (stop > text && space((unsigned char)stop[-1])) stop--;
        char *end;
        value = PyOS_string_to_double(text, &end, NULL);
        if (PyErr_Occurred()) {
            if (!PyErr_ExceptionMatches(PyExc_ValueError)) goto error;
            PyErr_Clear();
            value = Py_NAN;
        } else if (end != stop) {
            value = Py_NAN;
        }
store_value:
        memcpy((char *)output.buf + index * sizeof(double), &value, sizeof(double));
        index++;
        Py_CLEAR(item);
    }
    if (PyErr_Occurred()) goto error;
    if (index != output.len / (Py_ssize_t)sizeof(double)) {
        PyErr_SetString(PyExc_ValueError, "Output length differs from input length");
        goto error;
    }
    Py_DECREF(iterator);
    PyBuffer_Release(&output);
    Py_RETURN_NONE;
error:
    Py_XDECREF(item);
    Py_XDECREF(iterator);
    PyBuffer_Release(&output);
    return NULL;
}

static PyMethodDef methods[] = {
    {"parse_numeric", parse_numeric, METH_VARARGS, "Convert decimal strings into an existing float64 vector."},
    {"filter_pval", filter_pval, METH_VARARGS, "Filter a plain whitespace table by reference SNP before numeric parsing."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_input", NULL, -1, methods};
PyMODINIT_FUNC PyInit__input(void) { return PyModule_Create(&module); }
