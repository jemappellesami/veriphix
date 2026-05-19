OPENQASM 2.0;
include "qelib1.inc";
qreg q793[5];
cx q793[4],q793[3];
cx q793[2],q793[3];
cx q793[2],q793[1];
cx q793[1],q793[0];
rx(pi/4) q793[1];
