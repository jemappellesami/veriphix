OPENQASM 2.0;
include "qelib1.inc";
qreg q195[3];
rx(5*pi/4) q195[2];
rz(3*pi/4) q195[2];
rx(pi) q195[2];
cx q195[1],q195[2];
cx q195[1],q195[0];
