OPENQASM 2.0;
include "qelib1.inc";
qreg q487[3];
rx(pi) q487[2];
cx q487[1],q487[2];
cx q487[0],q487[1];
rx(pi/4) q487[1];
