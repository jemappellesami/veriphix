OPENQASM 2.0;
include "qelib1.inc";
qreg q388[4];
cx q388[0],q388[1];
rz(5*pi/4) q388[3];
cx q388[3],q388[2];
cx q388[2],q388[1];
cx q388[0],q388[1];
