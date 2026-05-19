OPENQASM 2.0;
include "qelib1.inc";
qreg q144[6];
cx q144[3],q144[4];
cx q144[2],q144[3];
cx q144[1],q144[2];
cx q144[0],q144[1];
rx(pi/4) q144[1];
