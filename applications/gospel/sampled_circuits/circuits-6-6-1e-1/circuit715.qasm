OPENQASM 2.0;
include "qelib1.inc";
qreg q716[6];
cx q716[3],q716[4];
cx q716[3],q716[2];
cx q716[1],q716[2];
cx q716[0],q716[1];
rx(pi/4) q716[1];
