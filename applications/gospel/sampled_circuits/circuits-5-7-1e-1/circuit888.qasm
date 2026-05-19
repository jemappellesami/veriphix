OPENQASM 2.0;
include "qelib1.inc";
qreg q889[5];
rx(pi) q889[4];
cx q889[4],q889[3];
cx q889[3],q889[2];
cx q889[1],q889[2];
cx q889[1],q889[0];
