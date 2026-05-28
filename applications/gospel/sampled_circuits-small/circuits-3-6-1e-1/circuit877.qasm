OPENQASM 2.0;
include "qelib1.inc";
qreg q878[3];
cx q878[0],q878[1];
cx q878[1],q878[2];
rx(pi) q878[0];
cx q878[1],q878[0];
rx(pi/4) q878[1];
