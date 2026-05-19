OPENQASM 2.0;
include "qelib1.inc";
qreg q254[5];
cx q254[3],q254[4];
cx q254[2],q254[3];
cx q254[1],q254[2];
cx q254[1],q254[0];
rx(pi/4) q254[1];
