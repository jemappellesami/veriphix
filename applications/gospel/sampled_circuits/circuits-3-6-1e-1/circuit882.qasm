OPENQASM 2.0;
include "qelib1.inc";
qreg q883[3];
rx(pi/2) q883[2];
cx q883[1],q883[2];
cx q883[0],q883[1];
rx(pi/4) q883[1];
