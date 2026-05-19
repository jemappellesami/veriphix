OPENQASM 2.0;
include "qelib1.inc";
qreg q922[5];
cx q922[3],q922[4];
cx q922[2],q922[3];
cx q922[2],q922[1];
cx q922[0],q922[1];
rx(pi/4) q922[1];
