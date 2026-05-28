OPENQASM 2.0;
include "qelib1.inc";
qreg q423[3];
cx q423[0],q423[1];
rx(pi/2) q423[1];
cx q423[1],q423[2];
cx q423[0],q423[1];
rx(pi/4) q423[1];
