OPENQASM 2.0;
include "qelib1.inc";
qreg q22[5];
cx q22[3],q22[4];
cx q22[3],q22[2];
cx q22[2],q22[1];
cx q22[1],q22[0];
rx(pi/4) q22[1];
