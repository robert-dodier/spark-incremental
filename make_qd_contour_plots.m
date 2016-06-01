load -ascii "qd_10pct_only.contour-data"
x_10pct_only = qd_10pct_only(1,2:52);
y_10pct_only = qd_10pct_only(2:52,1);
z_10pct_only = qd_10pct_only(2:52,2:52);

load -ascii "qd_10pct_plus_90pct.contour-data"
x_10pct_plus_90pct = qd_10pct_plus_90pct(1,2:52);
y_10pct_plus_90pct = qd_10pct_plus_90pct(2:52,1);
z_10pct_plus_90pct = qd_10pct_plus_90pct(2:52,2:52);

load -ascii "qd_90pct.contour-data"
x_90pct = qd_90pct(1,2:52);
y_90pct = qd_90pct(2:52,1);
z_90pct = qd_90pct(2:52,2:52);

load -ascii "qd_all.contour-data"
x_all = qd_all(1,2:52);
y_all = qd_all(2:52,1);
z_all = qd_all(2:52,2:52);

contour(x_10pct_only, y_10pct_only, z_10pct_only)
axis("equal")
print qd_10pct_only.svg

contour(x_10pct_plus_90pct, y_10pct_plus_90pct, z_10pct_plus_90pct)
axis("equal")
print qd_10pct_plus_90pct.svg

contour(x_90pct, y_90pct, z_90pct)
axis("equal")
print qd_90pct.svg

contour(x_all, y_all, z_all)
axis("equal")
print qd_all.svg

