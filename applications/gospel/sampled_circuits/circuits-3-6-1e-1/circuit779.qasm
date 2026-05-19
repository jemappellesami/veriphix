OPENQASM 2.0;
include "qelib1.inc";
qreg q780[3];
cx q780[1],q780[2];
rx(pi/4) q780[1];
cx q780[1],q780[0];
rx(pi/4) q780[1];
