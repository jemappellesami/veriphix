OPENQASM 2.0;
include "qelib1.inc";
qreg q945[3];
rx(pi/2) q945[0];
rx(3*pi/2) q945[2];
rz(pi/2) q945[2];
cx q945[2],q945[1];
cx q945[1],q945[0];
