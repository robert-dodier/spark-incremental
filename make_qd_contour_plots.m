load -ascii "qd_1pct_only.contour-data"
x_1pct_only = qd_1pct_only(1,2:52);
y_1pct_only = qd_1pct_only(2:52,1);
z_1pct_only = qd_1pct_only(2:52,2:52);

load -ascii "qd_1pct_plus_99pct.contour-data"
x_1pct_plus_99pct = qd_1pct_plus_99pct(1,2:52);
y_1pct_plus_99pct = qd_1pct_plus_99pct(2:52,1);
z_1pct_plus_99pct = qd_1pct_plus_99pct(2:52,2:52);

load -ascii "qd_99pct.contour-data"
x_99pct = qd_99pct(1,2:52);
y_99pct = qd_99pct(2:52,1);
z_99pct = qd_99pct(2:52,2:52);

load -ascii "qd_all.contour-data"
x_all = qd_all(1,2:52);
y_all = qd_all(2:52,1);
z_all = qd_all(2:52,2:52);

contour(x_1pct_only, y_1pct_only, z_1pct_only)
axis("equal")
print qd_1pct_only.svg

contour(x_1pct_plus_99pct, y_1pct_plus_99pct, z_1pct_plus_99pct)
axis("equal")
print qd_1pct_plus_99pct.svg

contour(x_99pct, y_99pct, z_99pct)
axis("equal")
print qd_99pct.svg

contour(x_all, y_all, z_all)
axis("equal")
print qd_all.svg

