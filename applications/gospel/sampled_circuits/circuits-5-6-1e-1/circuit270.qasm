OPENQASM 2.0;
include "qelib1.inc";
qreg q271[5];
cx q271[4],q271[3];
cx q271[2],q271[3];
cx q271[2],q271[1];
cx q271[1],q271[0];
rx(pi/4) q271[1];
