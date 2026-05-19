OPENQASM 2.0;
include "qelib1.inc";
qreg q797[5];
cx q797[4],q797[3];
cx q797[2],q797[3];
cx q797[1],q797[2];
cx q797[1],q797[0];
rx(pi/4) q797[1];
