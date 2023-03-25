/*********************************************
 * OPL 22.1.0.0 Model
 * Author: Santu Mondal
 * Creation Date: 21-Nov-2022 at 11:30:00 AM
 *********************************************/

int m = ...; // Number of Markets

range M = 1..m; // Set of Markets

int n = ...; // Number of Products

range K = 1..n; // Set of Products

tuple edge { // Edge Structure

  int i;
  int j;
}

{edge} E = {<i,j> | ordered i,j in M: i != j}; // Set of Edges

int c[E] = ...; // Cost Matrix
/*
dvar boolean x[1..m][1..m]; // Travel Decision Variable

//dvar boolean y[1..m];

int p[1..m][1..n] = ...; // Product Price Matrix

int q[1..m][1..n] = ...; // Product Availibility Matrix

dvar int z[1..m][1..n]; // Purchase Decision Variable 

int d[1..n] = ...; // Product Demand

minimize (
  sum (i in 1..m, j in 1..m) c[i][j] * x[i][j]
  +
  sum (i in 1..m, k in 1..n) p[i][k] * z[i][k]
);

subject to { // Constraints

  forall (i in 1..m) x[i][i] == 0;

  forall (k in 1..n) sum (i in 1..m) z[i][k] == d[k];

  forall (i in 1..m, k in 1..n) z[i][k] <= q[i][k];

  forall (i in 1..m) sum (j in 1..m: i != j) x[i][j] >= 0;
  forall (i in 1..m) sum (j in 1..m: i != j) x[i][j] <= 1;

  forall (j in 1..m) sum (i in 1..m: i != j) x[i][j] >= 0;
  forall (j in 1..m) sum (i in 1..m: i != j) x[i][j] <= 1;

  sum (i in 1..m) x[i][1] == 1;

  sum (j in 1..m) x[1][j] == 1;

  forall (i in 1..m, k in 1..n) z[i][k] >= 0;

//  forall (i in 1..m, j in 1..m: i != j) (x[i][j] + x[j][i]) >= 0;
//  forall (i in 1..m, j in 1..m: i != j) (x[i][j] + x[j][i]) <= 1;

}
*/
execute OUTPUT {

  writeln("\nOUTPUT\n======\n");

  writeln(c);

/*
  var cost = 0, tour = new Array();

  for (var i = 1, j = 0, pos = 0; j != 1; i = j) {
    tour[pos++] = i;
    for (j = 0; x[i][++j] == 0; );
    cost += c[i][j];
  }
  writeln("Tour: [", tour.join(", "), "]\n\nCost: ", cost);

  writeln("\n", pos);

  var t = 0, u = 0, t_cost = 0, p_cost = 0;

  for (var i = 1; i <= m; i++) {
    for (var j = 1; j <= m; j++) {
      if (x[i][j] == 1) {
        writeln(i, " -> ", j, x[j][i] == 1 ? " (?) " : " ");
        t += 1;
        t_cost += c[i][j] * x[i][j];
      }        
    }
    for (var k = 1; k <= n; k++) {
      p_cost += p[i][k] * z[i][k];
    }
  }
  write(t, " [");
  for (var i = 1; i <= m; i++) {
    if (y[i] == 1) {
      u += 1;
      write(i, ", ");
    }
  }
  writeln(" ], ", u);
  writeln("\nTotal Cost: ", (t_cost + p_cost), " (Travel Cost: ", t_cost, " + Purchase Cost: ", p_cost, ")");
*/
}
