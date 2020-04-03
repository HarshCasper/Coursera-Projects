#include <algorithm>
#include <iostream>
#include <vector>
#include <cstdio>

using namespace std;

typedef vector<double> Column;
typedef vector<double> Row;
typedef vector<Row> Matrix;


struct Position {
	Position(int column, int row):
		column(column),
		row(row)
	{}

	int column;
	int row;
};

static void print_ab(const char *s, const Matrix &a, const Column &b)
{
	printf("%s:\n", s); // alex
	for (int i = 0; i < a.size(); i++) {
		for (int j = 0; j < a[i].size(); j++) {
			printf("%.3f ", a[i][j]); // alex
		}
		printf("| %.3f\n", b[i]); // alex
	}
	printf("------------------\n"); // alex
}

static void print_vec(const char *s, const vector<double> &vec)
{
	printf("%s: ", s); // alex
	for (int i = 0; i < vec.size(); i++) {
		printf("%.3f ", vec[i]); // alex
	}
	printf("\n"); // alex
}

static vector<vector<int> > combination(int n, int k)
{
	vector<bool> bitmask(k, true);
	vector<vector<int> > combos;

	bitmask.insert(bitmask.end(), n - k, false);

	do {
		combos.push_back(vector<int>());

		for (int i = 0; i < n; i++) {
			if (bitmask[i]) {
				combos.back().push_back(i);
			}
		}
	} while (prev_permutation(bitmask.begin(), bitmask.end()));

	return combos;
}

static Position SelectPivotElement(const Matrix &a, int k)
{
	int size = a.size();
	double max_a = -numeric_limits<double>::max();

	Position pivot_element(k, 0);

	for (int i = k; i < size; i++) {
		if (std::abs(a[i][k]) > max_a) {
			max_a = std::abs(a[i][k]);
			pivot_element.row = i;
		}
	}

	if (std::abs(max_a) == 0) {
		return Position(-1, -1);
	}

	return pivot_element;
}

static int SwapLines(Matrix &a, Column &b, Position &pivot_element)
{
	if (pivot_element.column >= a.size()) return 1;
	if (pivot_element.row >= a.size()) return 1;

	swap(a[pivot_element.column], a[pivot_element.row]);
	swap(b[pivot_element.column], b[pivot_element.row]);
	pivot_element.row = pivot_element.column;

	return 0;
}

static void ProcessPivotElement(Matrix &a, Column &b, const Position &pivot_element)
{
	int size = a.size();
	int k = pivot_element.row;
	double f;

	f = a[k][k];
	a[k][k] = 1;

	for (int j = k + 1; j < size; j++) {
		a[k][j] /= f;
	}
	b[k] /= f;

	for (int i = 0; i < size; i++) {
		if (i == k) continue;

		f = a[i][k];

		for (int j = k + 1; j < size; j++) {
			a[i][j] -= a[k][j] * f;
		}
		a[i][k] = 0;

		b[i] -= b[k] * f;
	}
}


static pair<int, vector<double>> solve_diet_problem(int n, int m, const Matrix &a, const vector<double> &b, const vector<double> &c)
{
	const double BIG_NUMBER = 10e9;
	const double EPS = 1e-3;
	vector<double> solution;
	int solution_type = -1;
	Matrix a_big(n + m + 1, vector<double>(m, 0));
	Column b_big(n + m + 1);

	for (size_t i = 0; i < n; i++) {
		a_big[i] = a[i];
		b_big[i] = b[i];
	}

	for (size_t i = n; i < n + m; i++) {
		a_big[i][i - n] = -1;
		b_big[i] = 0;
	}

	for (size_t i = 0; i < m; i++) {
		a_big[n + m][i] = 1;
	}
	b_big[n + m] = BIG_NUMBER;

	vector<vector<int> > combos = combination(n + m + 1, m);
	double max_score = -numeric_limits<double>::max();

	for (size_t i = 0; i < combos.size(); i++) {
		Matrix a_temp(m, vector<double>(m, 0));
		Column b_temp(m);
		bool is_infinity = false;
		bool failed = false;

		for (size_t j = 0; j < m; j++) {
			int index = combos[i][j];

			a_temp[j] = a_big[index];
			b_temp[j] = b_big[index];

			if (b_temp[j] == BIG_NUMBER) {
				is_infinity = true;
			}
		}

		for (int j = 0; j < m; j++) {
			Position pivot_element = SelectPivotElement(a_temp, j);

			if (pivot_element.row == -1) {
				failed = true;
				break;
			}

			if (SwapLines(a_temp, b_temp, pivot_element)) {
				failed = true;
				break;
			}

			ProcessPivotElement(a_temp, b_temp, pivot_element);
		}
		if (failed) continue;

		vector<double> x = b_temp;

		for (int j = 0; j < n + m + 1; j++) {
			double sum = 0;

			for (int k = 0; k < m; k++) {
				sum += x[k] * a_big[j][k];
			}
			if (sum > b_big[j] + EPS) {
				failed = true;
				break;
			}
		}
		if (failed) continue;

		double score = 0;

		for (int j = 0; j < m; j++) {
			score += c[j] * x[j];
		}

		if (score > max_score) {
			max_score = score;
			solution = x;
			solution_type = is_infinity ? 1 : 0;
		}
	}

	return {solution_type, solution};
}

int main()
{
	int n, m;
	cin >> n >> m;
	Matrix a(n, vector<double>(m));
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			cin >> a[i][j];
		}
	}
	vector<double> b(n);
	for (int i = 0; i < n; i++) {
		cin >> b[i];
	}
	vector<double> c(m);
	for (int i = 0; i < m; i++) {
		cin >> c[i];
	}

	pair<int, vector<double>> ans = solve_diet_problem(n, m, a, b, c);

	switch (ans.first) {
		case -1: 
			printf("No solution\n");
			break;
		case 0: 
			printf("Bounded solution\n");
			for (int i = 0; i < m; i++) {
				printf("%.18f%c", ans.second[i], " \n"[i + 1 == m]);
			}
			break;
		case 1:
			printf("Infinity\n");
			break;      
	}
	return 0;
}
