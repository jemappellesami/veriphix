OPENQASM 2.0;
include "qelib1.inc";
qreg q77[3];
cx q77[1],q77[2];
rx(pi/2) q77[2];
cx q77[1],q77[2];
cx q77[1],q77[0];
rx(pi/4) q77[1];
