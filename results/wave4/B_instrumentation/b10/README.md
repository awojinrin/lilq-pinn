# B10: a note on the boundary weights

`bc_weight_total` (in `b10.csv`) and `b10_bc_weight_total` /
`paper_grid_bc_weight_total` (in `b10_vs_paper_grid.csv`) give the total
weight of one velocity component's boundary rows, in units of lambda_bc:

- equal weights (`B10-EQ`, and the paper grid): each edge's rows carry
  lambda_bc, so 4 lambda_bc over the four edges;
- Clenshaw-Curtis weights (`B10-CC`): the whole perimeter carries lambda_bc,
  as Addendum v2.3 Section 1 specifies.

So `B10-CC` and `B10-EQ` differ in total boundary weight as well as in the
quadrature (the advisor's reply on wave 3, item 2.8). The pressure pin is
lambda_bc in both.
