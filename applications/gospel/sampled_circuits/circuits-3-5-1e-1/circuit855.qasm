OPENQASM 2.0;
include "qelib1.inc";
qreg q856[3];
rz(7*pi/4) q856[2];
rx(pi) q856[2];
cx q856[2],q856[1];
cx q856[1],q856[0];
