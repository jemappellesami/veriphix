OPENQASM 2.0;
include "qelib1.inc";
qreg q1000[3];
rx(pi) q1000[2];
cx q1000[2],q1000[1];
cx q1000[1],q1000[0];
