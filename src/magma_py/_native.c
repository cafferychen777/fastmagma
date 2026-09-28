#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

/* QUADPACK owns no data: each capsule retains its immutable copied spectrum
 * until its LowLevelCallable is released. No Python API is used in callbacks. */
/* The cosine and sine quadratures revisit many identical nonnegative nodes.
 * Cache both components by exact double bits. Collisions only recompute values;
 * they never interpolate or change QUADPACK's error estimates. */
#define CACHE_SIZE 4096

typedef struct {
    uint64_t key;
    double cosine;
    double sine;
} IntegralNode;

typedef struct {
    unsigned int references;
    IntegralNode nodes[CACHE_SIZE];
} IntegralCache;

typedef struct {
    IntegralCache *cache;
    Py_ssize_t size;
    double parameter;
    int sine;
    double values[1];
} IntegralContext;

static const char *callback_signature = "double (double, void *)";

static void free_context(PyObject *capsule) {
    IntegralContext *context = PyCapsule_GetContext(capsule);
    if (context != NULL) {
        if (context->cache != NULL && --context->cache->references == 0) {
            PyMem_Free(context->cache);
        }
        PyMem_Free(context);
    }
}

static double tilted_component(double v, void *data) {
    const IntegralContext *context = (const IntegralContext *)data;
    IntegralNode *node = NULL;
    uint64_t key = 0;
    if (context->cache != NULL) {
        memcpy(&key, &v, sizeof(key));
        key += 1;  /* Nonnegative finite nodes never collide with the empty key. */
        uint64_t hash = key;
        hash ^= hash >> 33;
        hash *= UINT64_C(0xff51afd7ed558ccd);
        hash ^= hash >> 33;
        node = &context->cache->nodes[hash & (CACHE_SIZE - 1)];
        for (unsigned int probe = 0; probe < 8; ++probe) {
            IntegralNode *candidate = &context->cache->nodes[(hash + probe) & (CACHE_SIZE - 1)];
            if (candidate->key == key) {
                return context->sine ? candidate->sine : candidate->cosine;
            }
            if (candidate->key == 0) {
                node = candidate;
                break;
            }
        }
    }
    double phase = 0.0;
    double log_amplitude = 0.0;
    const double h = context->parameter;
    for (Py_ssize_t i = 0; i < context->size; ++i) {
        const double bv = context->values[i] * v;
        phase += atan(bv);
        log_amplitude += log1p(bv * bv);
    }
    phase *= 0.5;
    const double s = sin(phase);
    const double c = cos(phase);
    if (node == NULL) {
        const double numerator = context->sine ? h * s - v * c : h * c + v * s;
        return exp(-0.25 * log_amplitude) * numerator / (h * h + v * v);
    }
    const double amplitude = exp(-0.25 * log_amplitude);
    const double denominator = h * h + v * v;
    const double cosine = amplitude * (h * c + v * s) / denominator;
    const double sine = amplitude * (h * s - v * c) / denominator;
    if (node != NULL) {
        node->key = key;
        node->cosine = cosine;
        node->sine = sine;
    }
    return context->sine ? sine : cosine;
}

static double imhof_component(double u, void *data) {
    const IntegralContext *context = (const IntegralContext *)data;
    double phase = 0.0;
    double log_rho = 0.0;
    if (u == 0.0) {
        for (Py_ssize_t i = 0; i < context->size; ++i) {
            phase += context->values[i];
        }
        return 0.5 * (phase - context->parameter);
    }
    for (Py_ssize_t i = 0; i < context->size; ++i) {
        const double au = context->values[i] * u;
        phase += atan(au);
        log_rho += log1p(au * au);
    }
    return sin(0.5 * (phase - context->parameter * u)) * exp(-0.25 * log_rho) / u;
}

static PyObject *make_callback(PyObject *values, double parameter, int sine, int tilted) {
    Py_buffer buffer;
    if (PyObject_GetBuffer(values, &buffer, PyBUF_FORMAT | PyBUF_C_CONTIGUOUS) < 0) {
        return NULL;
    }
    if (buffer.ndim != 1 || buffer.itemsize != sizeof(double) ||
        strcmp(buffer.format, "d") != 0 || buffer.len == 0) {
        PyBuffer_Release(&buffer);
        PyErr_SetString(PyExc_ValueError, "Expected a nonempty contiguous float64 vector");
        return NULL;
    }
    if (!isfinite(parameter) || (tilted && parameter <= 0.0)) {
        PyBuffer_Release(&buffer);
        PyErr_SetString(PyExc_ValueError, "Invalid integration parameter");
        return NULL;
    }
    IntegralContext *context = PyMem_Malloc(sizeof(IntegralContext) + buffer.len - sizeof(double));
    if (context == NULL) {
        PyBuffer_Release(&buffer);
        return PyErr_NoMemory();
    }
    context->cache = NULL;
    context->size = buffer.len / sizeof(double);
    context->parameter = parameter;
    context->sine = sine;
    memcpy(context->values, buffer.buf, buffer.len);
    PyBuffer_Release(&buffer);
    for (Py_ssize_t i = 0; i < context->size; ++i) {
        if (!isfinite(context->values[i]) || context->values[i] < 0.0) {
            PyMem_Free(context);
            PyErr_SetString(PyExc_ValueError, "Spectrum must be finite and nonnegative");
            return NULL;
        }
    }
    PyObject *capsule = PyCapsule_New(
        (void *)(tilted ? tilted_component : imhof_component), callback_signature, NULL);
    if (capsule == NULL) {
        PyMem_Free(context);
        return NULL;
    }
    if (PyCapsule_SetContext(capsule, context) < 0) {
        PyMem_Free(context);
        Py_DECREF(capsule);
        return NULL;
    }
    PyCapsule_SetDestructor(capsule, free_context);
    return capsule;
}

static PyObject *tilted_integrand(PyObject *self, PyObject *args) {
    PyObject *values;
    double h;
    int sine;
    if (!PyArg_ParseTuple(args, "Odp", &values, &h, &sine)) {
        return NULL;
    }
    return make_callback(values, h, sine, 1);
}

static PyObject *tilted_integrands(PyObject *self, PyObject *args) {
    PyObject *values;
    double h;
    if (!PyArg_ParseTuple(args, "Od", &values, &h)) {
        return NULL;
    }
    PyObject *cosine = make_callback(values, h, 0, 1);
    if (cosine == NULL) {
        return NULL;
    }
    PyObject *sine = make_callback(values, h, 1, 1);
    if (sine == NULL) {
        Py_DECREF(cosine);
        return NULL;
    }
    IntegralCache *cache = PyMem_Calloc(1, sizeof(IntegralCache));
    if (cache == NULL) {
        Py_DECREF(cosine);
        Py_DECREF(sine);
        return PyErr_NoMemory();
    }
    cache->references = 2;
    ((IntegralContext *)PyCapsule_GetContext(cosine))->cache = cache;
    ((IntegralContext *)PyCapsule_GetContext(sine))->cache = cache;
    PyObject *result = PyTuple_Pack(2, cosine, sine);
    Py_DECREF(cosine);
    Py_DECREF(sine);
    return result;
}

static PyObject *imhof_integrand(PyObject *self, PyObject *args) {
    PyObject *values;
    double q;
    if (!PyArg_ParseTuple(args, "Od", &values, &q)) {
        return NULL;
    }
    return make_callback(values, q, 0, 0);
}

static PyMethodDef methods[] = {
    {"tilted_integrands", tilted_integrands, METH_VARARGS,
     "Create cosine/sine callbacks sharing a bounded exact-node cache."},
    {"tilted_integrand", tilted_integrand, METH_VARARGS,
     "Create an owning SciPy capsule for a tilted Fourier component."},
    {"imhof_integrand", imhof_integrand, METH_VARARGS,
     "Create an owning SciPy capsule for the ordinary Imhof integrand."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_native", "Native quadrature callbacks.", -1, methods
};

PyMODINIT_FUNC PyInit__native(void) {
    return PyModule_Create(&module);
}
