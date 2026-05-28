OPENQASM 2.0;
include "qelib1.inc";
qreg q707[3];
cx q707[2],q707[1];
rx(pi/2) q707[2];
cx q707[2],q707[1];
cx q707[0],q707[1];
rx(pi/4) q707[1];
