OPENQASM 2.0;
include "qelib1.inc";
qreg q560[3];
rx(pi/4) q560[0];
cx q560[1],q560[0];
cx q560[2],q560[1];
rx(pi/4) q560[0];
