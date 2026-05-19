OPENQASM 2.0;
include "qelib1.inc";
qreg q242[4];
cx q242[1],q242[2];
rx(pi) q242[2];
cx q242[1],q242[2];
cx q242[1],q242[0];
cx q242[3],q242[2];
