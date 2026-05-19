OPENQASM 2.0;
include "qelib1.inc";
qreg q363[4];
cx q363[1],q363[0];
cx q363[3],q363[2];
cx q363[0],q363[1];
cx q363[2],q363[1];
rx(pi/4) q363[0];
