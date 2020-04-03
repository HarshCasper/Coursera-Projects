#include <cmath>
#include <iostream>
#include <vector>
#include <limits>

const double EPS = 1e-6;
const int PRECISION = 20;

typedef std::vector<double> Column;
typedef std::vector<double> Row;
typedef std::vector<Row> Matrix;

struct Equation {
	Equation(const Matrix &a, const Column &b):
		a(a),
		b(b)
	{}

	Matrix a;
	Column b;
};

struct Position {
	Position(int column, int row):
		column(column),
		row(row)
	{}

	int column;
	int row;
};

Equation ReadEquation()
{
	int size;
	std::cin >> size;
	Matrix a(size, std::vector <double> (size, 0.0));
	Column b(size, 0.0);
	for (int row = 0; row < size; ++row) {
		for (int column = 0; column < size; ++column)
			std::cin >> a[row][column];
		std::cin >> b[row];
	}
	return Equation(a, b);
}

Position SelectPivotElement(const Matrix &a, int k)
{
	int size = a.size();
	double max_a = -std::numeric_limits<double>::max();

	Position pivot_element(k, 0);

	for (int i = k; i < size; i++) {
		if (std::abs(a[i][k]) > max_a) {
			max_a = std::abs(a[i][k]);
			pivot_element.row = i;
		}
	}

	return pivot_element;
}

void SwapLines(Matrix &a, Column &b, Position &pivot_element)
{
	std::swap(a[pivot_element.column], a[pivot_element.row]);
	std::swap(b[pivot_element.column], b[pivot_element.row]);
	pivot_element.row = pivot_element.column;
}

void ProcessPivotElement(Matrix &a, Column &b, const Position &pivot_element)
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

static void print_ab(const Matrix &a, const Column &b)
{
	for (int i = 0; i < a.size(); i++) {
		for (int j = 0; j < a[i].size(); j++) {
			printf("%.3f ", a[i][j]); // alex
		}
		printf("| %.3f\n", b[i]); // alex
	}
	printf("------------------\n"); // alex
}

Column SolveEquation(Equation equation)
{
	Matrix &a = equation.a;
	Column &b = equation.b;
	int size = a.size();

	for (int k = 0; k < size; k++) {
		Position pivot_element = SelectPivotElement(a, k);
		SwapLines(a, b, pivot_element);
		ProcessPivotElement(a, b, pivot_element);
	}

	return b;
}

void PrintColumn(const Column &column)
{
	int size = column.size();
	std::cout.precision(PRECISION);
	for (int row = 0; row < size; ++row)
		std::cout << column[row] << std::endl;
}

int main()
{
	Equation equation = ReadEquation();
	Column solution = SolveEquation(equation);
	PrintColumn(solution);
	return 0;
}
