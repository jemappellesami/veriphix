OPENQASM 2.0;
include "qelib1.inc";
qreg q77[4];
rx(pi) q77[3];
cx q77[2],q77[3];
cx q77[1],q77[2];
cx q77[0],q77[1];
