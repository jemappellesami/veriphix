OPENQASM 2.0;
include "qelib1.inc";
qreg q467[3];
cx q467[2],q467[1];
rz(5*pi/4) q467[1];
cx q467[1],q467[0];
rx(pi/4) q467[1];
