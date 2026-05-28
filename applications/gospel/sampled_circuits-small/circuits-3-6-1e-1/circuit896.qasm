OPENQASM 2.0;
include "qelib1.inc";
qreg q897[3];
rx(pi) q897[2];
cx q897[1],q897[2];
cx q897[0],q897[1];
rx(pi/4) q897[1];
