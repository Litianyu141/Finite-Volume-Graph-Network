/*
#################################################################################################################################
# All the source files in `minisat` folder were initially copied and later modified from https://github.com/feiwang3311/minisat #
# (which was taken from the MiniSat source at https://github.com/niklasso/minisat). The MiniSAT license is below.               #
#################################################################################################################################
/***********************************************************************************[SolverTypes.h]
Copyright (c) 2003-2006, Niklas Een, Niklas Sorensson
Copyright (c) 2007-2010, Niklas Sorensson

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
**************************************************************************************************/

#include <errno.h>
#include <zlib.h>

#include "minisat/utils/System.h"
#include "minisat/utils/ParseUtils.h"
#include "minisat/utils/Options.h"
#include "minisat/core/Dimacs.h"
#include "minisat/simp/SimpSolver.h"
#include "minisat/gym/GymSolver.h"

using namespace Minisat;

// Graph-Q-SAT UPD: Add additional parameters for the environment generation.
GymSolver::GymSolver(char* satProb, bool with_restarts, int max_decision_cap) {
	IntOption    verb   ("MAIN", "verb",   "Verbosity level (0=silent, 1=some, 2=more).", 0, IntRange(0, 2));
	S.verbosity = verb;
	S.with_restarts = with_restarts;
    S.max_decision_cap = max_decision_cap;
	gzFile in = gzopen(satProb, "rb");
    if (in == NULL) {
    	printf("ERROR! Could not open file: %s\n", satProb); 
    	exit(1);
    }
    parse_DIMACS(in, S, true);
    gzclose(in);

    S.eliminate(true);
    if (!S.okay()){
    	printf("ERROR! SAT problem from file: %s is UNSAT by simplification\n", satProb);
    	exit(1);
    }    
    
    // Comments by Fei: Now the solveLimited() function really just initialize the problem. It needs steps to finish up!
    vec<Lit> dummy;
    S.solveLimited(dummy);

}

void GymSolver::step(int decision) {
    if (decision == 32767) {
        S.agent_decision = S.default_pickLit(); // if the decision is MaxInt number, let the minisat decide!
    } else {
    	S.agent_decision = mkLit(abs(decision)-1, decision < 0);
    }
	S.step();
}

double GymSolver::getReward() {
	return S.env_reward;
}

bool GymSolver::getDone() {
	return !S.env_hold;
}

// Graph-Q-SAT UPD: Get metadata/clauses/activities helpers.
std::vector<int>* GymSolver::getMetadata() {
    return &S.env_state_metadata;
}

std::vector<int>* GymSolver::getAssignments() {
    return &S.env_state_assignments;
}

std::vector<std::vector<int> >* GymSolver::getClauses() {
    return &S.env_state_clauses;
}

std::vector<double>* GymSolver::getActivities() {
    return &S.env_state_activities;
}
