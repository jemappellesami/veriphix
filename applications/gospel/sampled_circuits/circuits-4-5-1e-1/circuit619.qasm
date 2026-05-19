OPENQASM 2.0;
include "qelib1.inc";
qreg q620[4];
rx(3*pi/2) q620[3];
cx q620[2],q620[3];
cx q620[2],q620[1];
cx q620[1],q620[0];
