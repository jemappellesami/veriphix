OPENQASM 2.0;
include "qelib1.inc";
qreg q66[4];
rx(pi/4) q66[0];
cx q66[3],q66[2];
cx q66[0],q66[1];
cx q66[1],q66[2];
rx(pi/4) q66[0];
