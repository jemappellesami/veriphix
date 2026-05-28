OPENQASM 2.0;
include "qelib1.inc";
qreg q263[3];
rx(5*pi/4) q263[2];
cx q263[2],q263[1];
cx q263[0],q263[1];
rx(pi/4) q263[1];
