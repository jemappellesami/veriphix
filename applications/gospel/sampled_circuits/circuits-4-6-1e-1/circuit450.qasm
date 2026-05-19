OPENQASM 2.0;
include "qelib1.inc";
qreg q451[4];
rx(pi/2) q451[3];
cx q451[3],q451[2];
cx q451[1],q451[2];
cx q451[1],q451[0];
rx(pi/4) q451[1];
