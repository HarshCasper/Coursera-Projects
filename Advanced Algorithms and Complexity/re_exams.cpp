#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cassert>
using namespace std;

static void explore(int x, const vector<vector<int> > &adj, vector<int> &visited, vector<int> &ordered)
{
	visited[x] = 1;

	const vector<int> &neibs = adj[x];

	for (size_t i = 0; i < neibs.size(); i++) {
		int y = neibs[i];

		if (!visited[y]) {
			explore(y, adj, visited, ordered);
		}
	}

	ordered.push_back(x);
}

static vector<int> topo_sort(const vector<vector<int> > &adj)
{
	vector<int> visited(adj.size(), 0);
	vector<int> ordered;

	for (size_t i = 0; i < adj.size(); i++) {
		int x = i;

		if (!visited[x]) {
			explore(x, adj, visited, ordered);
		}
	}

	reverse(ordered.begin(), ordered.end());

	return ordered;
}

static vector<vector<int> > get_reverse_graph(const vector<vector<int> > &adj)
{
	vector<vector<int> > r_adj(adj.size());

	for (size_t i = 0; i < adj.size(); i++) {
		int x = i;

		for (size_t j = 0; j < adj[i].size(); j++) {
			int y = adj[i][j];

			r_adj[y].push_back(x);
		}
	}

	return r_adj;
}

static void explore2(int x, const vector<vector<int> > &adj, vector<int> &visited, vector<int> &scc)
{
	const vector<int> &neibs = adj[x];

	visited[x] = 1;
	scc.push_back(x);

	for (size_t i = 0; i < neibs.size(); i++) {
		int y = neibs[i];

		if (!visited[y]) {
			explore2(y, adj, visited, scc);
		}
	}
}

static vector<vector<int> > strongly_connected_components(const vector<vector<int> > &adj)
{
	vector<vector<int> > r_adj = get_reverse_graph(adj);
	vector<int> ordered = topo_sort(r_adj);
	vector<int> visited(adj.size(), 0);
	vector<vector<int> > sccs;

	for (size_t i = 0; i < ordered.size(); i++) {
		int x = ordered[i];

		if (!visited[x]) {
			vector<int> scc;

			explore2(x, adj, visited, scc);
			sccs.push_back(scc);
		}
	}

	return sccs;
}

struct Clause {
	Clause() : firstVar(0), secondVar(0) {}
	Clause(int _firstVar, int _secondVar) : firstVar(_firstVar), secondVar(_secondVar) {}

	int firstVar;
	int secondVar;
};

struct TwoSatisfiability {
	int numVars;
	vector<Clause> clauses;

	TwoSatisfiability(int n, int m) :
		numVars(n),
		clauses(m)
	{  }

	int varToGraphIndex(int x)
	{
		return x > 0 ? x - 1 : -x - 1 + numVars;
	}

	int graphIndexToVar(int i)
	{
		return i < numVars ? i + 1 : -i - 1 + numVars;
	}
	bool isSatisfiable(vector<int> &result)
	{
		int n = numVars;
		vector<vector<int> > adj(2*n);
		vector<int> parent(2*n);

		/* Construct the Implication Graph G */
		for (size_t i = 0; i < clauses.size(); i++) {
			int x = clauses[i].firstVar;
			int y = clauses[i].secondVar;

			adj[varToGraphIndex(-x)].push_back(varToGraphIndex(y));
			adj[varToGraphIndex(-y)].push_back(varToGraphIndex(x));
		}

		/* Find Strongly Connected Components (SCCs) of G */
		vector<vector<int> > sccs = strongly_connected_components(adj);

		for (size_t i = 0; i < sccs.size(); i++) {
			const vector<int> &scc = sccs[i];

			for (size_t j = 0; j < scc.size(); j++) {
				parent[scc[j]] = i;
			}
		}

		/* Check for unsatisfiable */
		for (size_t i = 0; i < n; i++) {
			if (parent[i] == parent[i+n]) return false;
		}
		/* Topological Sort of SCCs */
		vector<vector<int> > scc_adj(sccs.size());

		for (size_t i = 0; i < adj.size(); i++) {
			const vector<int> &neibs = adj[i];
			int parent1 = parent[i];

			for (size_t j = 0; j < neibs.size(); j++) {
				int parent2 = parent[neibs[j]];

				if (parent1 == parent2) continue;
				scc_adj[parent1].push_back(parent2);
			}
		}

		vector<int> ordered = topo_sort(scc_adj);

		/* Reverse iteration of SCCs and assignment of variables */
		for (vector<int>::reverse_iterator rit = ordered.rbegin(); rit != ordered.rend(); rit++) {
			int parent = *rit;
			const vector<int> &scc = sccs[parent];

			for (size_t i = 0; i < scc.size(); i++) {
				int x = graphIndexToVar(scc[i]);

				if (result[scc[i] % n] == -1) {
					result[scc[i] % n] = x < 0 ? 1 : 0;
				}
			}
		}
		return true;
	}
};


/*
Arguments:
 * `n` - the number of vertices.
 * `edges` - list of edges, each edge is a pair (u, v), 1 <= u, v <= n.
 * `colors` - string consisting of `n` characters, each belonging to the set {'R', 'G', 'B'}.
 Return value:
 * If there exists a proper recoloring, return value is a string containing new colors, similar to the `colors` argument.
 * Otherwise, return value is an empty string.
 */
string assign_new_colors(int n, vector<pair<int, int>> edges, string colors)
{
	string ans = colors;
	vector<Clause> clauses;

	for (int i = 0; i < n; i++) {
		int x0, x1, x2;

		if (colors[i] == 'R') {
			x0 = 3*i + 1;
			x1 = 3*i + 2;
			x2 = 3*i + 3;
		} else if (colors[i] == 'G') {
			x0 = 3*i + 2;
			x1 = 3*i + 1;
			x2 = 3*i + 3;
		} else {
			x0 = 3*i + 3;
			x1 = 3*i + 1;
			x2 = 3*i + 2;
		}

		clauses.push_back(Clause(-x0, -x0));
		clauses.push_back(Clause(x1, x2));
		clauses.push_back(Clause(-x1, -x2));
	}

	for (size_t i = 0; i < edges.size(); i++) {
		pair<int, int> &edge = edges[i];
		int u = edge.first;
		int v = edge.second;
		int xur = 3*u + 1;
		int xug = 3*u + 2;
		int xub = 3*u + 3;
		int xvr = 3*v + 1;
		int xvg = 3*v + 2;
		int xvb = 3*v + 3;

		clauses.push_back(Clause(-xur, -xvr));
		clauses.push_back(Clause(-xug, -xvg));
		clauses.push_back(Clause(-xub, -xvb));
	}

	TwoSatisfiability twoSat(3*n, clauses.size());
	twoSat.clauses = clauses;

	vector<int> result(n, -1);
	if (twoSat.isSatisfiable(result)) {
		for (int i = 0; i < n; i++) {
			if (result[3*i] == 0) {
				ans[i] = 'R';
			} else if (result[3*i + 1] == 0) {
				ans[i] = 'G';
			} else if (result[3*i + 2] == 0) {
				ans[i] = 'B';
			} else {
				assert(0);
			}
		}
	} else {
		return "";
	}
}

int main()
{
	int n, m;
	cin >> n >> m;
	string colors;
	cin >> colors;
	vector<pair<int, int> > edges;
	for (int i = 0; i < m; i++) {
		int u, v;
		cin >> u >> v;
		edges.push_back(make_pair(u, v));
	}
	string new_colors = assign_new_colors(n, edges, colors);
	if (new_colors.empty()) {
		cout << "Impossible";
	} else {
		cout << new_colors << endl;
	}
}
