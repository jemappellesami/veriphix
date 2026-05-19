OPENQASM 2.0;
include "qelib1.inc";
qreg q911[3];
cx q911[1],q911[0];
cx q911[2],q911[1];
rx(pi) q911[0];
cx q911[1],q911[0];
rx(pi/4) q911[1];
