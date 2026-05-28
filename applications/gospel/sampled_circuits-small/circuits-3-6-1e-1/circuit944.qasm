OPENQASM 2.0;
include "qelib1.inc";
qreg q945[3];
rx(7*pi/4) q945[2];
cx q945[1],q945[2];
cx q945[0],q945[1];
rx(pi/4) q945[1];
