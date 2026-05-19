OPENQASM 2.0;
include "qelib1.inc";
qreg q223[3];
rx(3*pi/4) q223[1];
rx(7*pi/4) q223[2];
cx q223[2],q223[1];
cx q223[1],q223[0];
