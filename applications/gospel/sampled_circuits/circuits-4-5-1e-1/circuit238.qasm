OPENQASM 2.0;
include "qelib1.inc";
qreg q239[4];
rx(pi) q239[0];
cx q239[1],q239[0];
cx q239[1],q239[2];
cx q239[1],q239[0];
