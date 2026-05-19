OPENQASM 2.0;
include "qelib1.inc";
qreg q793[4];
rx(pi/4) q793[3];
rz(pi) q793[3];
cx q793[3],q793[2];
cx q793[2],q793[1];
cx q793[0],q793[1];
