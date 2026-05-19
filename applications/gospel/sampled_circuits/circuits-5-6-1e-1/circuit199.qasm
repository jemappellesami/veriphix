OPENQASM 2.0;
include "qelib1.inc";
qreg q200[5];
cx q200[4],q200[3];
cx q200[3],q200[2];
cx q200[2],q200[1];
cx q200[0],q200[1];
rx(pi/4) q200[1];
