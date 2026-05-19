OPENQASM 2.0;
include "qelib1.inc";
qreg q466[5];
rx(5*pi/4) q466[4];
cx q466[4],q466[3];
cx q466[3],q466[2];
cx q466[1],q466[2];
cx q466[1],q466[0];
