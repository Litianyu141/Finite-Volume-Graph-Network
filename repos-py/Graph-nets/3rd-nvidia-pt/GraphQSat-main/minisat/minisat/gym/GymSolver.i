/*
#################################################################################################################################
# All the source files in `minisat` folder were initially copied and later modified from https://github.com/feiwang3311/minisat #
# (which was taken from the MiniSat source at https://github.com/niklasso/minisat). The MiniSAT license is below.               #
#################################################################################################################################
*/

/*
This is the main interface for python to communicate with the solver. In the original implementation
the state was turned into a string which was later parsed on the python side.
We have rewritten this and return python lists instead. One step further would be to return numpy arrays, which
should potentially be faster
*/

%module GymSolver

%include <stdint.i>
// Graph-Q-SAT UPD: return vectors to support variable sized lists. The original version had a fixed sized arrays.
%typemap(out) std::vector<int> *getMetadata %{
    $result = PyList_New($1->size());
    for (int i = 0; i < $1->size(); ++i) {
    PyList_SetItem($result, i, PyLong_FromLong((*$1)[i]));
   }
%}

%typemap(out) std::vector<int> *getAssignments %{
    $result = PyList_New($1->size());
    for (int i = 0; i < $1->size(); ++i) {
    PyList_SetItem($result, i, PyLong_FromLong((*$1)[i]));
   }
%}

%typemap(out) std::vector<double> *getActivities %{
    $result = PyList_New($1->size());
    for (int i = 0; i < $1->size(); ++i) {
    PyList_SetItem($result, i, PyFloat_FromDouble((*$1)[i]));
   }
%}

%typemap(out) std::vector<std::vector <int> > *getClauses %{
    $result = PyList_New($1->size());
    for (int i = 0; i < $1->size(); ++i) {
        std::vector <int> curr_clause_vec = (*$1)[i];
        int clause_size = curr_clause_vec.size();

        PyObject* curr_clause = PyList_New(clause_size);
        for (int j = 0; j < clause_size; ++j) {
            PyList_SetItem(curr_clause, j, PyLong_FromLong(curr_clause_vec[j]));
        }
        PyList_SetItem($result, i, curr_clause);
    }
%}
// end of Graph-Q-SAT UPD.
%include "GymSolver.h"

%{
#include <zlib.h>
#include "GymSolver.h"
%}
