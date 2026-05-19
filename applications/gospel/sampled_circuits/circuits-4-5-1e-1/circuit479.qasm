OPENQASM 2.0;
include "qelib1.inc";
qreg q480[4];
rz(pi) q480[0];
cx q480[2],q480[3];
cx q480[1],q480[0];
cx q480[2],q480[1];
rx(pi/4) q480[0];
