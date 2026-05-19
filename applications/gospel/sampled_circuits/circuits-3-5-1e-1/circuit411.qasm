OPENQASM 2.0;
include "qelib1.inc";
qreg q412[3];
rx(pi/2) q412[0];
rx(5*pi/4) q412[2];
cx q412[2],q412[1];
cx q412[1],q412[0];
