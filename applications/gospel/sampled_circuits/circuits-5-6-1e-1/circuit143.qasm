OPENQASM 2.0;
include "qelib1.inc";
qreg q144[5];
cx q144[3],q144[4];
cx q144[3],q144[2];
cx q144[1],q144[2];
cx q144[1],q144[0];
rx(pi/4) q144[1];
