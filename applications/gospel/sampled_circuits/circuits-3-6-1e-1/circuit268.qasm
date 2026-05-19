OPENQASM 2.0;
include "qelib1.inc";
qreg q269[3];
rx(5*pi/4) q269[2];
cx q269[2],q269[1];
cx q269[0],q269[1];
rx(pi/4) q269[1];
