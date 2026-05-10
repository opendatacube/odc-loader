# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Helper methods for working with AWS
"""

import json
import os
import time
from typing import Any
from urllib.request import urlopen

import botocore
import botocore.session
from botocore.credentials import Credentials, ReadOnlyCredentials
from botocore.session import Session

__all__ = (
    "ec2_metadata",
    "ec2_current_region",
    "botocore_default_region",
    "auto_find_region",
    "get_creds_with_retry",
    "mk_boto_session",
    "get_aws_settings",
)


def _fetch_text(url: str, timeout: float = 0.1) -> str | None:
    try:
        with urlopen(url, timeout=timeout) as resp:
            if 200 <= resp.getcode() < 300:
                return resp.read().decode("utf8")
            return None
    except OSError:
        return None


def ec2_metadata(timeout: float = 0.1) -> dict[str, Any] | None:
    """
    Grab EC2 instance metadata.

    When running inside AWS returns dictionary describing instance identity.
    Returns None when not inside AWS
    """

    txt = _fetch_text(
        "http://169.254.169.254/latest/dynamic/instance-identity/document", timeout
    )

    if txt is None:
        return None

    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return None


def ec2_current_region() -> str | None:
    """Returns name of the region  this EC2 instance is running in."""
    cfg = ec2_metadata()
    if cfg is None:
        return None
    return cfg.get("region", None)


def botocore_default_region(session: Session | None = None) -> str | None:
    """Returns default region name as configured on the system."""
    if session is None:
        session = botocore.session.get_session()
    return session.get_config_variable("region")


def auto_find_region(session: Session | None = None, default: str | None = None) -> str:
    """
    Try to figure out which region name to use

    1. Region as configured for this/default session
    2. Region this EC2 instance is running in
    3. Value supplied in `default`
    4. raise exception
    """
    region_name = botocore_default_region(session)

    if region_name is None:
        region_name = ec2_current_region()

    if region_name is not None:
        return region_name

    if default is None:
        raise ValueError("Region name is not supplied and default can not be found")

    return default


def get_creds_with_retry(
    session: Session, max_tries: int = 10, sleep: float = 0.1
) -> Credentials | None:
    """Attempt to obtain credentials upto `max_tries` times with back off
    :param session: botocore session, see mk_boto_session
    :param max_tries: number of attempt before failing and returing None
    :param sleep: number of seconds to sleep after first failure (doubles on every consecutive failure)
    """
    for i in range(max_tries):
        if i > 0:
            time.sleep(sleep)
            sleep = min(sleep * 2, 10)

        creds = session.get_credentials()
        if creds is not None:
            return creds

    return None


def mk_boto_session(
    profile: str | None = None,
    creds: ReadOnlyCredentials | None = None,
    region_name: str | None = None,
) -> Session:
    """Get botocore session with correct `region` configured

    :param profile: profile name to lookup
    :param creds: Override credentials with supplied data
    :param region_name: default region_name to use if not configured for a given profile
    """
    session = botocore.session.Session(profile=profile)

    if creds is not None:
        session.set_credentials(creds.access_key, creds.secret_key, creds.token)

    _region = session.get_config_variable("region")
    if _region is None:
        if region_name is None or region_name == "auto":
            _region = auto_find_region(session, default="us-west-2")
        else:
            _region = region_name
        session.set_config_variable("region", _region)

    return session


def aws_unsigned_check_env() -> bool:
    def parse_bool(v: str) -> bool:
        return v.upper() in ("YES", "Y", "TRUE", "T", "1")

    for evar in ("AWS_UNSIGNED", "AWS_NO_SIGN_REQUEST"):
        v = os.environ.get(evar, None)
        if v is not None:
            return parse_bool(v)

    return False


def get_aws_settings(
    profile: str | None = None,
    region_name: str = "auto",
    aws_unsigned: bool | None = None,
    requester_pays: bool = False,
) -> tuple[dict[str, Any], Credentials | None]:
    """
    Compute ``aws=`` parameter for ``set_default_rio_config``.

    see also ``datacube.utils.rio.set_default_rio_config``

    Returns a tuple of: ``(aws: Dictionary, creds: session credentials from botocore)``.

    Note that credentials are baked in to ``aws`` setting dictionary,
    however since those might be STS credentials they might require refresh
    hence they are returned from this function separately as well.
    """
    session = mk_boto_session(profile=profile, region_name=region_name)

    region_name = session.get_config_variable("region")

    if aws_unsigned is None:
        aws_unsigned = aws_unsigned_check_env()

    if aws_unsigned:
        return ({"region_name": region_name, "aws_unsigned": True}, None)

    creds = get_creds_with_retry(session)
    if creds is None:
        raise ValueError("Couldn't get credentials")

    cc = creds.get_frozen_credentials()

    return (
        {
            "region_name": region_name,
            "aws_access_key_id": cc.access_key,
            "aws_secret_access_key": cc.secret_key,
            "aws_session_token": cc.token,
            "requester_pays": requester_pays,
        },
        creds,
    )
