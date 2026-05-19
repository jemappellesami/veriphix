OPENQASM 2.0;
include "qelib1.inc";
qreg q510[4];
rx(pi) q510[3];
rz(pi) q510[3];
cx q510[3],q510[2];
cx q510[1],q510[2];
cx q510[1],q510[0];
